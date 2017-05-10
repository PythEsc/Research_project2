package dataconverter;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

/**
 *
 * @author Tobi
 * @Author Pieter Schaap - p_je@hotmail.com
 *         Added Training and validation Separation.
 */
public class DataConverter {
	public static long tescoCount = 0;
	public static long sainsburyCount = 0;
	public static long alltescoCount = 0;
	public static HashMap<File, Integer> resultCountTesco = new HashMap<>();
	public static HashMap<File, Integer> resultCountSainsbury = new HashMap<>();
	public static long allsainsburyCount = 0;
	public static final String PREFIX_TESCO = "TE_";
	public static final String PREFIX_SAINSBURY = "SA_";
	public static final String[] IGNORE_LIST = { "January", "February" };
	public static final String PREFIX_FILE = "page";
	public static final String POSTFIX_FILE_COMMENT = "comments.tab";
	public static final String POSTFIX_FILE_FULLSTAT = "fullstats.tab";
	private static final char DEFAULT_SEPARATOR = ';';

	/** The ratio of the validation set. */
	public static final float VALIDATION_FACTOR = 0.1f;
	private static String outputFolderTesco = "Data" + File.separator + "Tesco";
	private static String outputFolderSainsbury = "Data" + File.separator + "Sainsbury";

	private static String firstLine_comments = "position	post_id	post_by	post_text	post_published	comment_id	comment_by	is_reply	comment_message	comment_published	comment_like_count";
	private static String firstLine_fullstats = "type	by	post_id	post_link	post_message	picture	full_picture	link	link_domain	post_published	post_published_unix	post_published_sql	likes_count_fb	comments_count_fb	reactions_count_fb	shares_count_fb	engagement_fb	comments_retrieved	comments_base	comments_replies	comment_likes_count	rea_NONE	rea_LIKE	rea_LOVE	rea_WOW	rea_HAHA	rea_SAD	rea_ANGRY	rea_THANKFUL";

	/**
	 * @param args
	 *            the command line arguments
	 */
	public static void main(String[] args) {

		Runnable runnableSains = () -> {
			int killerCounter = 0;
			String sainsburyPath = "Sainsbury_s";
			File sainsbury = new File(sainsburyPath);
			File outputSainsbury = new File(outputFolderSainsbury);
			if (!outputSainsbury.exists()) {
				outputSainsbury.mkdirs();
			}
			if (!outputSainsbury.exists()) {
				throw new IllegalStateException("Mkdirs failed for path: " + outputSainsbury.getAbsolutePath());
			}
			if (!outputSainsbury.canWrite()) {
				throw new IllegalStateException("You do not have the needed write permissions for: " + outputSainsbury.getAbsolutePath());
			}
			File[] listFiles = sainsbury.listFiles();
			for (File listFile : listFiles) {
				String name = listFile.getName();
				if (name.startsWith(PREFIX_SAINSBURY) && !name.endsWith(IGNORE_LIST[0]) && !name.endsWith(IGNORE_LIST[1])) {
					killerCounter += convertFolder(listFile, outputFolderSainsbury);
				}
			}
			System.out.println("Sainsbury: old size " + allsainsburyCount + " posts.");
			System.out.println("Sainsbury: survived " + killerCounter + " posts.");
			System.out.println("Sainsbury: removed " + sainsburyCount + " posts.");
		};

		Runnable runnableTesco = () -> {
			int killerCounter = 0;
			String tescozipPath = "Tesco";
			File tesco = new File(tescozipPath);
			File outputTesco = new File(outputFolderTesco);
			if (!outputTesco.exists()) {
				outputTesco.mkdirs();
			}
			if (!outputTesco.exists()) {
				throw new IllegalStateException("Mkdirs failed for path: " + outputTesco.getAbsolutePath());
			}
			if (!outputTesco.canWrite()) {
				throw new IllegalStateException("You do not have the needed write permissions for: " + outputTesco.getAbsolutePath());
			}
			File[] listFiles = tesco.listFiles();
			for (File listFile : listFiles) {
				String name = listFile.getName();
				if (name.startsWith(PREFIX_TESCO) && !name.endsWith(IGNORE_LIST[0]) && !name.endsWith(IGNORE_LIST[1])) {
					killerCounter += convertFolder(listFile, outputFolderTesco);
				}
			}
			System.out.println("Tesco: old size " + alltescoCount + " posts.");
			System.out.println("Tesco: survived " + killerCounter + " posts.");
			System.out.println("Tesco: removed " + tescoCount + " posts.");

			System.out.println("[TESCO] Selecting validation and training sets...");
			File trainFile = new File(outputTesco + File.separator + "training.txt");
			File valFile = new File(outputTesco + File.separator + "validation.txt");
			boolean errors = false;
			for (File f : outputTesco.listFiles()) {
				try {
					DataConverter.createValidationSet(resultCountTesco.get(f), f, trainFile, valFile);
				} catch (IOException e) {
					errors = true;
					e.printStackTrace();
				}
			}

			if (errors)
				System.err.println("One or more errors Occurred.");
		};

		Thread sains = new Thread(runnableSains);
		Thread tesc = new Thread(runnableTesco);

		sains.start();

		tesc.start();

	}

	/**
	 * Converts a comments and fullstats file of the facebook data from .tab file
	 * to .csv file with ";" as separators and filters its content.
	 *
	 * @param file
	 *            The current folder
	 * @param outputFolder
	 *            The outputfolder of Tesco or Sainsbury
	 *
	 * @return
	 */
	public static int convertFolder(File file, String outputFolder) {
		ArrayList<String> mapFullstatsIds = new ArrayList<>();
		File commentFile = null;
		File fullStatsFile = null;
		for (File listFile : file.listFiles()) {
			if (listFile.getName()
						.endsWith(POSTFIX_FILE_COMMENT)) {
				commentFile = listFile;
			} else if (listFile	.getName()
								.endsWith(POSTFIX_FILE_FULLSTAT)) {
				fullStatsFile = listFile;
			}
		}

		File parent = new File(outputFolder + "/" + file.getName());
		if (!parent.exists()) {
			parent.mkdir();
		}
		// Compute
		File fullstats = new File(outputFolder + "/" + file.getName() + "/fullstats.csv");

		checkValidIdsAndWriteLinesFullstats(fullStatsFile, fullstats, mapFullstatsIds, outputFolder);
		writeLinesComments(outputFolder, file, commentFile, mapFullstatsIds);
		return mapFullstatsIds.size();
	}

	private static void writeLinesComments(String outputFolder, File file, File commentFile, ArrayList<String> mapFullstatsIds) {
		try {
			File comments = new File(outputFolder + "/" + file.getName() + "/comments.csv");
			Writer commentsFW = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(comments), "UTF-8"));
			try (Scanner sc = new Scanner(commentFile, "UTF-8")) {
				sc.useLocale(new Locale("gb"));
				boolean firstLineFound = false;
				while (sc.hasNextLine()) {
					String nextLine = sc.nextLine();
					if (!firstLineFound && nextLine.equals(firstLine_comments)) {
						firstLineFound = true;
					} else if (firstLineFound) {
						String[] split = nextLine.split("\t");
						String id = split[1];
						ArrayList<String> listToAdd = new ArrayList<>();
						listToAdd.addAll(Arrays.asList(split));
						if (mapFullstatsIds.contains(id) && !listToAdd.isEmpty()) {
							writeLine(commentsFW, listToAdd);
						}
					}
				}
				commentsFW.close();
			}
		} catch (IOException ex) {
			Logger	.getLogger(DataConverter.class.getName())
					.log(Level.SEVERE, null, ex);
		}
	}

	private static void checkValidIdsAndWriteLinesFullstats(File fullStatsFile, File fullstats, ArrayList<String> mapFullstatsIds,
			String outputFolder) {
		try {
			try (Scanner sc = new Scanner(fullStatsFile, "UTF-8")) {
				Writer fullstatsFw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fullstats), "UTF-8"));
				sc.useLocale(new Locale("gb"));
				boolean firstLineFound = false;
				while (sc.hasNextLine()) {
					String nextLine = sc.nextLine();
					if (!firstLineFound && nextLine.equals(firstLine_fullstats)) {
						firstLineFound = true;
					} else if (firstLineFound) {
						if (outputFolder.equals(outputFolderSainsbury)) {
							allsainsburyCount++;
						} else {
							alltescoCount++;
						}
						String[] split = nextLine.split("\t");
						String text = split[4];
						if (text.trim()
								.length() > 20 && split[5]	.trim()
															.isEmpty()) {
							String id = split[2];
							mapFullstatsIds.add(id);
							ArrayList<String> listToAdd = new ArrayList<>();
							listToAdd.addAll(Arrays.asList(split));
							if (!listToAdd.isEmpty()) {
								writeLine(fullstatsFw, listToAdd);

							}
						} else {
							if (outputFolder.equals(outputFolderSainsbury)) {
								sainsburyCount++;
							} else {
								tescoCount++;
							}
						}
					}
				}
				fullstatsFw.close();
			}
		} catch (IOException ex) {
			Logger	.getLogger(DataConverter.class.getName())
					.log(Level.SEVERE, null, ex);
		}
	}

	public static void writeLine(Writer w, List<String> values) throws IOException {
		writeLine(w, values, DEFAULT_SEPARATOR, ' ');
	}

	private static String followCVSformat(String value) {

		String result = value;
		if (result.contains("\"")) {
			result = result.replace("\"", "\"\"");
		}
		if (result.contains(";")) {
			result = "\"" + result + "\"";
		}
		return result;

	}

	public static void writeLine(Writer w, List<String> values, char separators, char customQuote) throws IOException {

		boolean first = true;

		// default customQuote is empty
		if (separators == ' ') {
			separators = DEFAULT_SEPARATOR;
		}

		StringBuilder sb = new StringBuilder();
		for (String value : values) {
			if (!first) {
				sb.append(separators);
			}
			if (customQuote == ' ') {
				sb.append(followCVSformat(value));
			} else {
				sb	.append(customQuote)
					.append(followCVSformat(value))
					.append(customQuote);
			}

			first = false;
		}
		sb.append("\n");
		w.append(sb.toString());

	}

	/**
	 * Divides the dataset into a validation set and a training set.
	 * 
	 * @param dataSize
	 * @param origin
	 * @param trainingFile
	 * @param validationFile
	 * @throws IOException
	 */
	private static void createValidationSet(int dataSize, File origin, File trainingFile, File validationFile) throws IOException {

		// First we divide the data into 10 buckets in order to assure posts are selected from the entire time range.
		int stepSize = (int) VALIDATION_FACTOR * dataSize;

		ArrayList<Integer> selected = new ArrayList<>();

		// Select the posts
		for (int i = 0; i + stepSize <= dataSize; i += stepSize) {

			// Compute a random offset
			int index = i + ((int) (i + Math.random() * stepSize));

			// Assert that we don't go out of bounds.
			assert index < dataSize - 1 : "The validationset creator went out of bounds.";

			selected.add(index);
		}

		// Assure the list is sorted
		Collections.sort(selected);

		CSVFormat csvFormat = CSVFormat	.newFormat('\n')
										.withRecordSeparator(';');

		CSVParser parser = CSVParser.parse(origin, Charset.forName("UTF-8"), csvFormat);
		// Create appending file writer to collect multiple files into a single file.
		FileWriter valWriter = new FileWriter(validationFile, true), trainWriter = new FileWriter(trainingFile, true);
		CSVPrinter valPrinter = new CSVPrinter(valWriter, csvFormat), trainPrinter = new CSVPrinter(trainWriter, csvFormat);
		Iterator<Integer> selIt = selected.iterator();
		Iterator<CSVRecord> csvIterator = parser.iterator();

		int curLine = 0;
		while (selIt.hasNext()) {

			// The next index that is to be moved to the validation index.
			int nextValidationIndex = selIt.next();

			while (csvIterator.hasNext()) {
				CSVRecord rec = csvIterator.next();
				if (curLine == nextValidationIndex) {
					// Print the record in the validation set
					valPrinter.printRecord(rec);
				} else {
					// Print the data in the training set
					trainPrinter.printRecord(rec);
				}

			}


		}
		valPrinter.close();
		trainPrinter.close();
	}

}
